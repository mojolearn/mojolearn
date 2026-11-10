# SPDX-License-Identifier: Apache-2.0
"""Native constructor/parameter names and fitted-state metadata decisions."""


def _linear_mode_space(point: Int) -> Bool:
    # The whitespace accepted by CPython str.strip, including its four
    # information-separator controls. No Unicode lower-case mapping outside
    # ASCII maps to a letter in either supported mode word.
    return (9 <= point <= 13 or 28 <= point <= 32 or point == 0x85 or
        point == 0xA0 or point == 0x1680 or 0x2000 <= point <= 0x200A or
        point == 0x2028 or point == 0x2029 or point == 0x202F or
        point == 0x205F or point == 0x3000)


def linear_mode_name(source: String) raises -> String:
    var word = String()
    var started = False
    var ended = False
    for codepoint in source.codepoints():  # driver-loop: over public numeric-mode text
        var point = Int(codepoint)
        if _linear_mode_space(point):
            ended = started
            continue
        if ended or point >= 128:
            raise Error("numeric_mode must be fast, identical or None")
        started = True
        if 65 <= point <= 90:
            point += 32
        word += String(chr(point))
    if word != "identical" and word != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    return word^


def linear_vendor_code(name: String) -> Int:
    if name == "cpu":
        return 0
    if name == "metal":
        return 1
    if name == "cuda":
        return 2
    if name == "hip":
        return 3
    if name == "none":
        return 4
    return -1


def linear_vendor_name(code: Int) raises -> String:
    if code == 0:
        return "cpu"
    if code == 1:
        return "metal"
    if code == 2:
        return "cuda"
    if code == 3:
        return "hip"
    if code == 4:
        return "none"
    raise Error("linear metadata: unknown compiled vendor")


def linear_parameter_names(ridge: Bool) -> List[String]:
    if ridge:
        return ["alpha", "fit_intercept", "normalize", "numeric_mode", "solver"]
    return ["fit_intercept", "numeric_mode"]


def linear_constructor_names(ridge: Bool) -> List[String]:
    if ridge:
        return ["alpha", "solver", "fit_intercept", "normalize", "numeric_mode"]
    return ["fit_intercept", "numeric_mode"]


def linear_constructor_check(count: Int, ridge: Bool) raises:
    if count != (5 if ridge else 2):
        raise Error("TypeError: native linear constructor requires its complete configuration")


def linear_parameter_head(key: String) -> String:
    var separator = key.find("__")
    return String(key[byte=0:separator]) if separator >= 0 else String(key)


def linear_parameter_index(key: String, ridge: Bool) -> Int:
    var name = linear_parameter_head(key)
    var names = linear_parameter_names(ridge)
    for i in range(len(names)):  # driver-loop: over fixed constructor parameter names
        if names[i] == name:
            return i
    return -1


def linear_parameter_refusal(key: String, index: Int, shown: String,
    estimator: String, ridge: Bool) raises:
    if index < 0:
        var valid = "['alpha', 'fit_intercept', 'normalize', 'numeric_mode', 'solver']" if ridge else "['fit_intercept', 'numeric_mode']"
        raise Error("Invalid parameter " + shown + " for estimator " + estimator + ". Valid parameters are: " + valid + ".")
    if key.find("__") >= 0:
        raise Error("AttributeError: scalar parameter has no set_params")


def linear_public_fitted_key(key: String) -> Bool:
    return key.endswith("_") and not key.startswith("_")


def linear_fitted_names(ridge: Bool) -> List[String]:
    var names: List[String] = ["coef_", "intercept_", "_x_mean", "_y_mean", "n_features_in_", "input_copied_", "_mojolearn_fitted"]
    if ridge:
        names.append("solver_")
    return names^
