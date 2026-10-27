"""Source-reviewed forecast readbacks; never fit models during a smoke."""

THETA_MODELS = {
    "theta": "STM", "optimized-theta": "OTM", "dynamic-theta": "DSTM",
    "dynamic-optimized-theta": "DOTM", "auto-theta": None,
}
CROSTON = {"croston", "croston-optimized", "croston-sba"}


def statsforecast_record(model, bp):
    # FAIRNESS FIX (2026-10-13 audit): the recipe previously declared only
    # kwargs, omitting fixed model selection and constructor settings. Read
    # actual objects, including model on Theta subclasses whose signatures
    # omit that inherited attribute (statsforecast/models.py).
    record = bp.arm_record(model)
    for name, value in vars(model).items():
        if not name.startswith("_"):
            record["params"][name] = bp._jsonable(value)
    record["source"] += " + public constructor attributes"
    return record


def semantic_exception(lane, param, arm, ours, theirs, library):
    # FAIRNESS FIX (2026-10-13 audit): None was treated as an unspecified
    # default even where it explicitly selects fixed-order fitting or
    # automatic estimation. Every allowance is lane/library/value scoped.
    slug = lane.removeprefix("algos/")
    if lane == "algos/var" and library == "statsmodels" and param == "ic":
        if ("ic" in ours and "ic" in theirs and ours["ic"] is None
                and theirs["ic"] is None and ours.get("maxlags") == theirs.get("maxlags")
                and ours.get("maxlags") == 2):
            return ("ic=None explicitly disables order selection on both; maxlags=2 is the "
                    "fixed order (mojolearn _x_sequence_var.py; statsmodels vector_ar/var_model.py)")
    if not lane.startswith("algos/") or library != "statsforecast":
        return None
    if slug not in THETA_MODELS and slug not in CROSTON:
        return None
    if param == "alias" and (ours.get("alias") is None or isinstance(ours.get("alias"), str)):
        if isinstance(theirs.get("alias"), str):
            return "alias is a display label only (_x_sequence_croston.py; statsforecast models._TS.__repr__)"
    if param == "prediction_intervals" and param not in ours and param in theirs and theirs[param] is None:
        return "prediction_intervals=None disables conformal intervals; the recipe requests only mean forecasts"
    if (slug in CROSTON and param == "only_conformal_intervals" and param not in ours
            and theirs.get(param) is True and "prediction_intervals" in theirs
            and theirs["prediction_intervals"] is None):
        return "only_conformal_intervals selects the interval method; no intervals are requested and prediction_intervals=None"
    if slug in THETA_MODELS:
        if param == "distribution" and param not in ours and theirs.get(param) == "normal":
            return "normal is the only supported Theta error distribution in mojolearn; statsforecast AutoTheta uses normal, explicitly pinned for AutoTheta and fixed by its subclass super calls"
        if param in ("initial_smoothed", "alpha", "theta") and param in ours and ours[param] is None and param not in theirs:
            return ("None requests estimation: statsforecast models.AutoTheta calls theta.auto_theta "
                    "without this argument, whose fixed default is None; mojolearn _x_sequence_theta.py agrees")
        if param == "model" and slug == "auto-theta" and "model" in ours and "model" in theirs and ours[param] is None and theirs[param] is None:
            return "model=None explicitly searches STM/OTM/DSTM/DOTM on both AutoTheta implementations"
    return None
