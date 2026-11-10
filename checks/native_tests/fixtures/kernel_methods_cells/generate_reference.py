"""Independent fixture/reference generation; never imports or runs Mojolearn.

The first80rows reproduce test_km_kernels._data(n=80,d=6,seed=4).
One extra input row supports neighboring rectangular shapes. Reference cells
come from sklearn's float64 pairwise APIs on the recorded float32 inputs.
Independent reference matrices run in parallel; no product timing or identity.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import pairwise

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng = np.random.default_rng(4)
    original = np.abs(rng.normal(size=(81, 6))).astype(np.float32)
    original[::7, 2] = 0.0
    # Input compatibility with the former80-row Python numerical fixture.
    expected = np.abs(np.random.default_rng(4).normal(size=(80, 6))).astype(np.float32)
    expected[::7, 2] = 0.0
    assert np.array_equal(original[:80], expected)
    edges = np.array([
        [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
        [np.nextafter(np.float32(1),np.float32(2)), np.nextafter(np.float32(1),np.float32(0)),0,0],
        [1,2,4,8], [1e-12,2e-12,0,0]], dtype=np.float32)
    negative = np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype=np.float32)
    inputs = {"original":original,"edges":edges,"negative":negative}
    records = []
    for name, data in inputs.items():
        path=HERE/(name+"-input.f32")
        path.write_bytes(data.astype("<f4").tobytes())
        records.append({"path":path.name,"sha256":digest(path),"shape":list(data.shape),"dtype":"<f4"})
    jobs=[(name,kernel,gamma) for name in ("original","edges")
          for kernel,gamma in (("cosine",1.),("chi2",1.),("chi2-g03",.3),("additive-chi2",1.))]
    jobs.append(("negative","cosine",1.))

    def reference(job):
        name,kernel,gamma=job
        values=inputs[name].astype(np.float64)
        if kernel=="cosine": result=pairwise.cosine_similarity(values)
        elif kernel=="additive-chi2": result=pairwise.additive_chi2_kernel(values)
        else: result=pairwise.chi2_kernel(values,gamma=gamma)
        path=HERE/(name+"-"+kernel+".f64")
        path.write_bytes(result.astype("<f8").tobytes())
        return {"path":path.name,"sha256":digest(path),"shape":list(result.shape),"dtype":"<f8", "gamma":gamma}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        records.extend(pool.map(reference,jobs))
    metadata={"schema":1,"numpy":np.__version__,"scikit_learn":sklearn.__version__,
      "generator_sha256":digest(Path(__file__)),"pairwise_source_sha256":digest(Path(pairwise.__file__)),
      "origin":"test_km_kernels._data80x6 seed4; first80input rows byte-identical",
      "oracle":"sklearn pairwise float64 on recorded float32 inputs; no Mojolearn execution",
      "records":records,"nystroem":{"training_shape":[80,6],"components":12,"seed":1,"atol":.005,"rtol":1e-7},
      "parallel_workers":os.cpu_count()}
    (HERE/"reference.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"files":len(records),"numpy":np.__version__,"sklearn":sklearn.__version__,"product_execution":False}))


if __name__=="__main__":
    main()
