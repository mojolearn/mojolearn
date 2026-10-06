# SPDX-License-Identifier: Apache-2.0
"""Actual minibatch SGD fusion/OvR caller, canonical host replay.

Promoted defaults are unchanged. Independent OFF-arm builds attribute
batch-step fusion and independent OvR scheduling. Device qualification
is pending; compilation never establishes numerical identity or speed.
"""
from std.memory import bitcast
from x_linear.ops import FP,IP
from x_linear.team import solo
from x_linear.sgd import sgd_fit,L_HINGE,L_LOG,L_SQUARED,P_L2,P_EN,LR_CONSTANT,LR_PA1
from x_linear.device import _sgd_mb_grid,_sgd_mb_ovr_applies,_sgd_chunk

def fp(mut x: List[Float32]) -> FP:
    return x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
def ip(mut x: List[Int32]) -> IP:
    return x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

def check(n: Int,d: Int,k: Int,batch: Int,loss: Int,rate: Int,penalty: Int,weighted: Bool) raises:
    var problems=k if k>2 else 1
    var x=List[Float32](length=n*d,fill=Float32(0))
    var y=List[Float32](length=n*(2 if weighted else 1),fill=Float32(0))
    for i in range(n):
        y[i]=Float32((i*7)%k) if k>0 else Float32(i%7-3)*Float32(0.125)
        if weighted:
            y[n+i]=Float32((i%3)+1)*Float32(0.25)
        for j in range(d):
            x[i*d+j]=Float32((i*11+j*7)%31-15)*Float32(0.03125)
    var ints: List[Int32]=[Int32(k),Int32(loss),Int32(penalty),Int32(rate),1,5,2,1,123,0,Int32(1 if weighted else 0),0,Int32(batch),0]
    var floats: List[Float32]=[0.01,0.5,0.03125,0.5,0.1,-1]
    var out=problems*d+problems+2
    var expected=List[Float32](length=out,fill=Float32(0))
    var actual=List[Float32](length=out,fill=Float32(0))
    var work=List[Float32](length=problems*(n+d)+problems,fill=Float32(0))
    var order=List[Int32](length=problems*n,fill=Int32(0))
    var team_work=List[Float32](length=1,fill=Float32(0))
    sgd_fit(solo(fp(team_work),n,0,0),fp(x),fp(y),n,d,ip(ints),fp(floats),fp(expected),fp(work),ip(order))
    for repeat in range(2):
        _sgd_mb_grid(fp(x),n*d,fp(y),len(y),n,d,ints,floats,out,fp(actual))
        for i in range(out):
            if bitcast[DType.uint32](actual[i])!=bitcast[DType.uint32](expected[i]):
                print("I12 SGD mismatch n,d,k,batch,loss,rate,word",n,d,k,batch,loss,rate,i,bitcast[DType.uint32](actual[i]),bitcast[DType.uint32](expected[i]))
                raise Error("I12 SGD full coefficient/intercept/epoch/status identity failed")
    print("I12 SGD_CALL_PASS",n,d,k,batch,loss,rate,"ovr_admitted",_sgd_mb_ovr_applies(problems,n,d,batch,_sgd_chunk()))

def main() raises:
    check(257,7,3,16,L_HINGE,LR_CONSTANT,P_L2,False)
    check(513,33,5,64,L_LOG,LR_CONSTANT,P_EN,True)
    check(257,7,3,16,L_HINGE,LR_PA1,P_L2,True)
    check(257,7,2,16,L_LOG,LR_CONSTANT,P_L2,False)
    check(257,7,0,16,L_SQUARED,LR_CONSTANT,P_L2,False)
    print("I12 SGD_PASS actual_minibatch_fusion OVR_independence tail shuffled repeated_fits complete_words")
