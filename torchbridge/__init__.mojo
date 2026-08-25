# torchbridge: the PyTorch surface of `mojolearn.identical.gemm.fp32.v1`.
#
# NEVER COMPILED, NEVER EXECUTED. Nothing in this directory has been through
# the Mojo compiler or run on any device. See TORCH_BRIDGE_PLAN.md, which is
# the design and carries the full OWED list.
#
# This file is deliberately empty of declarations. `CustomOpLibrary` requires
# an `__init__.mojo` for the directory to be a Mojo package it can load
# (`max/develop/custom-kernels-pytorch`, "This file marks the folder as a Mojo
# package, which CustomOpLibrary requires"). Putting anything importable here
# would mean the kernel library's op discovery walks it, and the op discovery
# in `max/experimental/torch/torch.py::CustomOpLibrary.__init__` swallows the
# exception from every name it cannot register. A failure that is swallowed
# and stored in `_deferred_errors` is a failure nobody sees until they touch
# that exact attribute, so the fewer names in this package, the better.
#
# The ops are in `identical_ops.mojo`.
