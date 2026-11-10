# SPDX-License-Identifier: Apache-2.0
"""Native constructor/parameter names and fitted-state metadata decisions."""


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
