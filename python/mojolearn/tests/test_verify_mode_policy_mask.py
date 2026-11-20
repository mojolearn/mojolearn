"""verify refuses an identical set whose binary reports a relaxed mode policy row."""
from mojolearn import _verify


class _Binding(object):
    def __init__(self, mask):
        self._mask = mask

    def mojolearn_mode_policy_mask(self):
        return self._mask


def test_identical_set_with_no_row_relaxed_is_accepted():
    masks = {"_mojolearn": 0, "_mojolearn_gbdt": 0}
    assert _verify._mode_policy_problem("identical", masks) is None


def test_identical_set_with_a_planted_row_is_refused():
    problem = _verify._mode_policy_problem("identical", {"_mojolearn": 1, "_mojolearn_gbdt": 0})
    assert problem is not None
    assert "_mojolearn" in problem and "fold_free" in problem


def test_every_row_bit_is_named_in_the_refusal():
    for bit, row in enumerate(_verify._POLICY_ROWS):
        problem = _verify._mode_policy_problem("identical", {"_mojolearn_gbdt": 1 << bit})
        assert problem is not None and row in problem


def test_fast_set_reports_its_rows_without_a_refusal():
    assert _verify._mode_policy_problem("fast", {"_mojolearn": 79}) is None


def test_unreadable_mask_is_recorded_as_none_and_not_refused():
    assert _verify._read_policy_mask(None, "mojolearn_mode_policy_mask") is None
    assert _verify._read_policy_mask(object(), "mojolearn_mode_policy_mask") is None
    assert _verify._mode_policy_problem("identical", {"_mojolearn": None}) is None


def test_mask_is_read_from_the_binding_entry():
    assert _verify._read_policy_mask(_Binding(0), "mojolearn_mode_policy_mask") == 0
    assert _verify._read_policy_mask(_Binding(4), "mojolearn_mode_policy_mask") == 4


def test_mode_report_carries_the_masks_in_its_receipt():
    mode = _verify.ModeReport()
    mode.policy_masks = {"_mojolearn": 0}
    assert mode.as_dict()["policy_masks"] == {"_mojolearn": 0}
