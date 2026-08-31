# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`cuml/cpp/include/cuml/solvers/params.hpp`, transliterated.

Three enums. `cdFit` accepts only `loss_funct::SQRD_LOSS` (`cd.cuh:130`);
`lr_type` and `penalty` belong to the SGD solver (`sgd.cuh`) and are kept
here because the header is one file and the port is file for file.
"""

# enum lr_type { OPTIMAL, CONSTANT, INVSCALING, ADAPTIVE }
comptime LR_OPTIMAL = 0
comptime LR_CONSTANT = 1
comptime LR_INVSCALING = 2
comptime LR_ADAPTIVE = 3

# enum loss_funct { SQRD_LOSS, HINGE, LOG }
comptime LOSS_SQRD_LOSS = 0
comptime LOSS_HINGE = 1
comptime LOSS_LOG = 2

# enum penalty { NONE, L1, L2, ELASTICNET }
comptime PENALTY_NONE = 0
comptime PENALTY_L1 = 1
comptime PENALTY_L2 = 2
comptime PENALTY_ELASTICNET = 3


def loss_funct_name(loss: Int) -> String:
    if loss == LOSS_SQRD_LOSS:
        return String("SQRD_LOSS")
    if loss == LOSS_HINGE:
        return String("HINGE")
    if loss == LOSS_LOG:
        return String("LOG")
    return String("loss_funct(") + String(loss) + ")"
