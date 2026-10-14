#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SCORER PREFLIGHT, shared by the race drivers (lane harness-evidence, 2026-10-14).

Before any timed work a race proves its quality scorer can run: the reference data
it reads is present, the scorer imports, and it scores known data (a self-test on
a tiny slice). If not, every arm of the race is REFUSED with the reason
"quality scorer unavailable: ...", and nothing is timed. A timed cell whose scorer
then still fails is "not-scored" (tools/bench_board.py mark_not_scored), never "ok".

The drivers with their own references (tools/bench_board_algos.py: own-host and
scikit-learn oracles; tools/bench_board_neural.py: the host forward) run those
here-equivalent checks themselves; the classical drivers
(tools/classical_two_datasets.py, tools/bench_board_more.py) use this module.
No timing, no device work: NumPy on the conductor, before any worker starts.
"""
import json
import os

SCORER_REFUSED = "quality scorer unavailable: "


class ScorerUnavailable(RuntimeError):
    """A race's quality scorer cannot run: refuse the race before any timed work."""


class Preflight:
    """Collects checks into a receipt; the first failed check raises ScorerUnavailable."""

    def __init__(self, lane, dataset):
        self.receipt = {"lane": lane, "dataset": dataset, "checks": [], "status": "running"}

    def check(self, name, fn):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001
            self.receipt["checks"].append({"check": name, "status": "failed", "detail": repr(exc)[:600]})
            self.receipt["status"] = "refused"
            raise ScorerUnavailable("%s: %s" % (name, str(exc)[:600] or repr(exc)))
        self.receipt["checks"].append({"check": name, "status": "ok", "detail": detail})
        return detail

    def done(self):
        self.receipt["status"] = "ok"
        return self.receipt


def require_arrays(data, keys):
    """The arrays a scorer reads are present and non-empty in the race's data: {key: shape}."""
    import numpy as np
    absent = [k for k in keys if data is None or k not in data or data[k] is None or not np.size(data[k])]
    if absent:
        raise ValueError("the race data holds no %s" % ", ".join(absent))
    return {k: list(np.shape(data[k])) for k in keys}


def self_test(score, outputs, expect):
    """Run `score(outputs)` (the race's own scorer on known outputs) and require every arm's entry to carry
    one of the metrics in `expect` and no error. Returns the numeric metrics it produced."""
    got = score(outputs) or {}
    out = {}
    for arm in outputs:
        q = got.get(arm) or {}
        if q.get("error") or q.get("quality_error"):
            raise ValueError("the scorer failed on known data: %s" % (q.get("error") or q.get("quality_error")))
        if not any(k in q for k in expect):
            raise ValueError("the scorer produced none of %s on known data: %r" % (list(expect), q))
        out[arm] = {k: v for k, v in q.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return out


def refuse(result, arms, reason, out_json, log_prefix):
    """Mark every arm REFUSED with `reason` (no timing ran) and write the race JSON. Returns the driver rc (1)."""
    why = SCORER_REFUSED + reason if not str(reason).startswith(SCORER_REFUSED) else str(reason)
    result.setdefault("scorer_preflight", {}).update(status="refused", reason=why)
    result["quality"] = {"error": why, "scorer": "failed"}
    for arm in arms:
        result.setdefault("arms", {})[arm] = {"command": None, "warmup_ms": None, "ms": [], "digests": [],
                                              "mem": [], "status": "REFUSED", "error": why, "median_ms": None}
        print("%s-REFUSED lane=%s dataset=%s arm=%s stage=scorer-preflight detail=%s"
              % (log_prefix, result.get("lane"), result.get("dataset"), arm, json.dumps(why)), flush=True)
    with open(out_json + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True, default=str)
    os.replace(out_json + ".tmp", out_json)
    return 1
