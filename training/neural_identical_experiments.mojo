# SPDX-License-Identifier: Apache-2.0
"""NEURAL-only source experiments, 2026-10-06; no admitted performance claims.

Every switch is opt-in in IDENTICAL and disabled by MOJOLEARN_IDN_ALL_OFF.
These source hypotheses have not been compiled, tested, measured or qualified.
Same-version NVIDIA/AMD/Apple/host identity and whole-task quality remain gates.
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime NEURAL_IDN_EXPERIMENTS_ALLOWED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# A03: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. Preserve both
# rounded S20 activation and S21 product; backward and trace still own S20.
comptime NEURAL_SWIGLU_SAVE = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_SWIGLU_SAVE"
]()

# A04: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. An explicit
# byte-LM prefill owner discards carry storage; its per-layer stages survive.
comptime NEURAL_PREFILL_NO_CARRY = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_PREFILL_NO_CARRY"
]()

# A05: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. One serial
# row-stat owner, then block lanes apply cells; no reduction reassociation.
comptime NEURAL_RMS_APPLY_LANES = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_RMS_APPLY_LANES"
]()

# T05: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. Write the
# probability and its gradient in one launch with the original rounded seams.
comptime NEURAL_CE_FUSED_GRAD = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_CE_FUSED_GRAD"
]()

# T03: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. Rebind block
# weights and whole-write gradients to their owned arena slices each call.
comptime NEURAL_PARAM_VIEWS = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_PARAM_VIEWS"
]()

# T04: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. Upload one
# complete token batch to retained scratch, split inputs/targets on device.
comptime NEURAL_TOKEN_BATCH_UPLOAD = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_TOKEN_BATCH_UPLOAD"
]()

# V06: NOT TESTED — NOT COMPILED — NOT MEASURED. Default OFF. NEW VERSION:
# optimizer AdamW decay scalar = ftz(fma(-lr, weight_decay, +1)), replacing
# ftz(1 - ftz(mul(lr, weight_decay))). Shared host/device StepScalars only;
# new bits are allowed versus B, never among columns running this profile.
comptime NEURAL_ADAMW_DECAY_FMA = NEURAL_IDN_EXPERIMENTS_ALLOWED and is_defined[
    "MOJOLEARN_IDN_NEURAL_ADAMW_DECAY_FMA"
]()
