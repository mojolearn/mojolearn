# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/robust_native.mojo` (wave 8 run split): the RobustScaler fit's x_prep
stage program as typed Mojo. Imports no Python. The binding keeps UNPACK (the
fields, entries and the arena layout words), runs the program through the
shared transport and keeps PACK (the fitted arrays), as before."""
from core.preprocessing_program import prep_program_stage as _stage


def _robust_fit_program(mut program: List[Int32], select: Bool, blocked: Bool, unit: Bool, fused: Bool,
                        centering: Bool, scaling: Bool, n: Int, d: Int, blocks: Int, sorted: Int, part: Int,
                        stats: Int, fractions: Int, q: Int, center: Int, scale: Int, dest: Int):
    """sort_cols (unless the radix quantile `select`), the column statistics
    (2048-row blocks when `blocked`), the three quantiles, center and scale,
    the unit-variance rescale and, with `fused` (fit_transform), the affine
    stage into `dest`."""
    if not select:
        _stage(program,0,d,0,n,d,sorted,0)
    if blocked:
        _stage(program,124,blocks*d,0,n,d,part,blocks)
        _stage(program,125,d,part,blocks,d,stats)
    else:
        _stage(program,1,d,0,n,d,stats)
    _stage(program,2,3*d,0 if select else sorted,n,d,fractions,3,q,stats,1 if select else 0)
    _stage(program,4,d,q,3,d,center,scale,0,0,2,1)
    if unit:
        _stage(program,73,d,scale,fractions)
    if fused:
        _stage(program,3,n*d,0,n*d,d,center if centering else -1,scale if scaling else -1,dest)
