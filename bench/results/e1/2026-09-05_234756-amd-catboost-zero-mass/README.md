# Weighted CTR fixture attempt: RED

The revised learned-tree fixture still did not produce an occupied zero-mass
leaf. The optimizer was allowed to choose a shallower tree, so this did not
establish the intended leaf-estimation coverage. This failing attempt is
retained; the later fixed-partition production-estimator regression provides
the explicit zero-mass coverage while retaining learned-tree weighted checks.
