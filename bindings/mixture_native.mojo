# SPDX-License-Identifier: Apache-2.0
# The compiled owner of GaussianMixture (python/mojolearn/mixture.py) and BayesianGaussianMixture (_expansion_cluster.py) fit/predict/score: one native call per public method,
# reached from bindings/family_native.mojo for every `family_mix_*` operation.
from std.python import Python, PythonObject


def mixture_operation(model: PythonObject, operation: String, args: PythonObject,
         context: PythonObject) raises -> PythonObject:
    raise Error("RuntimeError: unknown family operation " + operation)
