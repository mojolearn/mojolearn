# SPDX-License-Identifier: Apache-2.0
# RUN part of bindings/native_scalar_promotion.mojo: typed Mojo only (no
# Python imports). The NumPy version text parse; the door reads the text from
# the admitted module namespace and keeps the promotion-state getter checks.


def native_numpy_version_weak(text: String, label: String) raises -> Bool:
    """True when the NumPy version `text` is 2.2 or later (weak promotion is
    then the only state); refused when it does not start with major.minor."""
    var major = 0
    var minor = 0
    var component = 0
    var digits = 0
    for codepoint in text.codepoints():  # driver-loop: bounded native version metadata
        var point = Int(codepoint)
        if point == 46:
            if digits == 0:
                raise Error("TypeError: " + label + " requires native NumPy version metadata")
            if component == 1:
                break
            component = 1
            digits = 0
        elif point >= 48 and point <= 57:
            if component == 0:
                major = min(100, major * 10 + point - 48)
            else:
                minor = min(100, minor * 10 + point - 48)
            digits += 1
        else:
            break
    if component == 0 or digits == 0:
        raise Error("TypeError: " + label + " requires native NumPy version metadata")
    return major > 2 or (major == 2 and minor >= 2)
