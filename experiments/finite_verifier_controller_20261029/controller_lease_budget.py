"""External finite-run admission only; never creates or renews cloud leases."""
import math

def lease_budget(deadline, unit_seconds, reserve_seconds, now):
    values = (deadline, unit_seconds, reserve_seconds)
    if all(x is None for x in values):
        return None
    if any(x is None for x in values):
        raise ValueError('deadline, unit budget and retention reserve must be supplied together')
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (*values, now)):
        raise ValueError('lease times and budgets must be positive finite numbers')
    remaining = deadline - now
    return dict(deadline_epoch=deadline, checked_epoch=now, remaining_seconds=remaining,
                unit_budget_seconds=unit_seconds, retention_reserve_seconds=reserve_seconds,
                allowed=remaining >= unit_seconds + reserve_seconds,
                limitation='Admission budget, not a runtime prediction or automatic lease extension')
