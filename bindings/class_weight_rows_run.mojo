# SPDX-License-Identifier: Apache-2.0
"""RUN of the class-weight row plan (bindings/class_weight_rows_boundary.mojo):
the typed plan built from admitted per-class weights, which the door runs with
the GIL released. No Python import (tools/python_free_dirs.sh). Lane
rs-shared-infra-b (2026-11-12): moved word for word; one weight per class,
nothing to split across cores."""
from core.class_weight_plan import ClassWeightInputs, ClassWeightPlan, class_weight_all_one, class_weight_plan


def class_weight_rows_plan(mut inputs: ClassWeightInputs, balanced: Bool) raises -> ClassWeightPlan:
    """`class_weight_plan_admitted`'s GIL-released step: for "balanced" the
    all-one flag of the computed ratios, then the per-class plan."""
    if balanced:
        inputs.all_one = class_weight_all_one(inputs.values)
    return class_weight_plan(inputs.values, inputs.all_one)
