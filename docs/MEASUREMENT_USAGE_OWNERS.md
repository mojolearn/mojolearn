# Measurement usage owners

`tools/runpod_usage_lease.py` and `tools/do_amd_usage_owner.py` accept
`idle_seconds: 1800`, `2700` or `3600`. The default remains 2700 seconds.
For the owner-requested one-hour retention after all queued campaign work,
set `idle_seconds: 3600` in the new frozen controller configuration. The
read-only `plan CONFIG` command reports the configured policy.

The idle interval starts only after work has finished and its capture is
verified. Pending-work holds block idle deletion. Resumed work, a changed
capture or capture failure clears/restarts idle eligibility; one completed
batch is not the end of a queue. RunPod's `hold` and `release-hold` clear the
previous idle/capture state. The AMD owner additionally checks the shared
queue; its local `HOLD` must be refreshed before its existing 30-minute TTL.
Maintain holds/queue state through every batch handoff.

This setting does not extend the separate orphan deadline: its default remains
5400 seconds (AMD renews 90 minutes). A healthy manager renews it; a missing
owner still triggers the independent orphan policy. Capture timeout remains
at most 1800 seconds. Job timeouts and existing remote guards are unchanged.

Do not edit a live adopted configuration in place: ownership binds its digest.
Coordinate an explicit controller handoff and preserve the prior state and
evidence. These tools do not by themselves prove complete dataset/model
retention; required bytes must be verified off-machine before teardown.
