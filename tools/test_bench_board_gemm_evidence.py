import importlib.util
from pathlib import Path
import numpy as np
import json


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_gemm_input_bytes_shapes_seed_and_real_callable_are_compared():
    N,P=load("bench_board_neural"),load("bench_board_params")
    def own(a,b): return a@b
    def other(a,b): return a@b
    own.__module__="mojolearn.linalg";other.__module__="torch"
    data=dict(a=np.ones((2,3),np.float32),b=np.ones((3,4),np.float32))
    a=N._gemm_record("gemm","mojolearn",own,data)
    b=N._gemm_record("gemm","torch",other,data)
    transported=json.loads(json.dumps({"ours":P.arm_record(a),"opponent":P.arm_record(b)}))
    report=P.check("gemm",transported)
    assert report["verdict"]=="MATCHED"
    assert report["arms"]["opponent"]["implementation"]["module"]=="torch"
    assert report["arms"]["ours"]["params"]["seed"]==7
    data["a"][0,0]=2
    changed=N._gemm_record("gemm","torch",other,data)
    assert P.check("gemm",dict(ours=a,opponent=changed))["verdict"]=="REFUSED"
