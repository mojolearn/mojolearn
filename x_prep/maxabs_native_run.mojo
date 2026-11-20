# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/maxabs_native.mojo` (wave 8 run split): the MaxAbsScaler fit's x_prep
stage program as typed Mojo. Imports no Python. The binding keeps UNPACK (the
fields, entries and the arena layout words), runs the program through the
shared transport and keeps PACK (the fitted arrays), as before."""
from core.preprocessing_program import prep_program_stage as _stage


def _maxabs_fit_program(mut program: List[Int32], blocked: Bool, work: Bool, fused: Bool, n: Int, d: Int,
                        blocks: Int, host: Int, sc: Int, dest: Int) -> Int:
    """The column maxima of |X| (2048-row blocks when `blocked`), the scale
    and, with `fused` (fit_transform), the affine stage into `dest`. Returns
    the max_abs_ offset; the scale is at `sc`."""
    var ma = 0
    if blocked:
        var part = host if work else sc+d
        ma = sc+d if work else part+5*blocks*d
        _stage(program,124,blocks*d,0,n,d,part,blocks)
        _stage(program,128,d,part,blocks,d,ma,sc)
    else:
        var st = sc+d
        ma = st+5*d
        _stage(program,1,d,0,n,d,st)
        _stage(program,4,d,ma,1,d,-1,sc,1)
    if fused:
        _stage(program,3,n*d,0,n*d,d,-1,sc,dest)
    return ma
