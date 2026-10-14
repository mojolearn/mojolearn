import copy

import recover_transform_inference as R


def test_only_same_scored_query_outputs_receive_retained_quality():
    fit = dict(arm="ours", hash="scored", quality={"rel_diff_vs_sklearn": .2},
               measurement=dict(id="fit", quality_recovery={"artifact": {"status": "verified"}}))
    inf = dict(arm="ours", batch="Xq", hash="scored", status="ok", quality={}, median_ms=7.,
               measurement={"id": "inference"})
    race = dict(cells=[fit], infer_cells=[inf])
    for field, value in [("hash", "different"), ("batch", "other"), ("quality", {"error": "existing"})]:
        altered = copy.deepcopy(race)
        altered["infer_cells"][0][field] = value
        assert R.attach(altered, {}) == 0
    assert R.attach(race, {"status": "verified"}) == 1
    assert inf["quality"] == fit["quality"]  # retain failures too
    assert inf["median_ms"] == 7.
    assert fit["measurement"]["id"] == "fit"
