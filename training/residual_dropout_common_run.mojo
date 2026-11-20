# SPDX-License-Identifier: Apache-2.0
"""RUN half of `bindings/residual_dropout_common.mojo` (wave 8 run split): the
typed residual-dropout arguments and the address checks shared by the host and
device layers. Typed Mojo only: no Python import."""

@fieldwise_init
struct ResidualDropoutArgs(Copyable,Movable):
    var n: Int
    var offset: Int
    var seed_lo: Int
    var seed_hi: Int
    var stream: Int
    var p: Float32


def _residual_overlap(a: Int,b: Int,n: Int) -> Bool:
    return a<b+4*n and b<a+4*n


def residual_dropout_addresses[BACKWARD: Bool](x: Int,residual: Int,first: Int,second: Int,n: Int) raises:
    if n==0:
        return
    if x==0 or first==0:
        raise Error("residual_dropout refuses null borrowed buffers")
    if _residual_overlap(x,first,n):
        raise Error("residual_dropout outputs must be disjoint from inputs")
    comptime if BACKWARD:
        if second==0 or _residual_overlap(second,x,n) or _residual_overlap(first,second,n):
            raise Error("residual_dropout backward outputs must be nonnull and disjoint")
    else:
        if residual==0 or _residual_overlap(first,residual,n):
            raise Error("residual_dropout residual/output must be nonnull and disjoint")
