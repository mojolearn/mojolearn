# SPDX-License-Identifier: Apache-2.0
"""The refusal of a GP query an entry flagged (RUN): typed on the status
words and the fold store only. The admitted query itself is built in the door
(`bindings/gp_query_x_door.mojo`)."""
from core.admitted_refusals import admitted_finite_refuse, door_rows_refuse
from bindings.device_rows_door_run import DoorRowsStore


def gp_query_refuse(finite: Int, in_range: Int, store: DoorRowsStore, est: String) raises:
    """The refusal of a query the entry flagged (its status words)."""
    if store.active():
        door_rows_refuse(finite, in_range, est, store.base_rows)
    else:
        admitted_finite_refuse(finite, est)
