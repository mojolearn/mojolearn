"""The route a classical row's scored call took, read AFTER the round's clock.

Lane flag-classical-dispatch-gate (2026-11-15). The SVC/SVR doors have one
route: the by-name entries with the device-resident fitted state (the B arm
`-D MOJOLEARN_CLASSICAL_NATIVE_DISPATCH`, the descriptor table, was removed
2026-11-23 as plumbing, and with it the `_native_svc_dispatch` field it
published). Every other classical
estimator has no classical dispatch switch on its path (identical x_neighbors
calls admit the native snapshot in every build), which is recorded by name.

Plain dictionary reads only: no fit, predict or binding call happens here, and
a read failure is recorded by name and never breaks a timing. Used by
tools/classical_two_datasets.py and tools/bench_board_more.py beside each
round's time (`classical_routes` in the arm record, ` route=` on the line).
"""

_SVM_CLASSES = ("SVC", "SVR", "NuSVC", "NuSVR")


def _model(runner):
    callback = getattr(runner, "model_for_hash", None)
    if callback is not None:
        try:
            return callback()
        except Exception:  # noqa: BLE001
            pass
    for name in ("est", "nn", "kd", "model", "trainer"):
        model = getattr(runner, name, None)
        if model is not None:
            return model
    return None


def classical_route(runner, model=None):
    """{"estimator", "svc_dispatch" | "classical_dispatch", "fields"} for
    our arm's model, or None when the runner holds no model (an opponent)."""
    model = model if model is not None else _model(runner)
    if model is None or type(model).__module__.split(".")[0] != "mojolearn":
        return None
    route = {"estimator": type(model).__name__}
    try:
        fields = vars(model)
        if type(model).__name__ in _SVM_CLASSES:
            route["svc_dispatch"] = "by-name"
        else:
            route["classical_dispatch"] = "none on this path"
        # Any route word a door publishes on the model (generic: *_route).
        for name, value in fields.items():
            if name.endswith("_route") and isinstance(value, (str, int, bool)):
                route.setdefault("fields", {})[name.lstrip("_")] = value
    except Exception as exc:  # noqa: BLE001
        route["error"] = type(exc).__name__ + ": " + str(exc)
    return route


def route_suffix(route):
    """` route=<word>` for a CTD/MORE round line, else empty."""
    if not isinstance(route, dict):
        return ""
    if "error" in route:
        return " route=unread"
    return " route=" + str(route.get("svc_dispatch") or route.get("classical_dispatch", "")).replace(" ", "-")
