# SPDX-License-Identifier: Apache-2.0
"""The parallel MLP parameter names and shapes, in their published order (RUN)."""


def _mlp_names() -> List[String]:
    var result = List[String]()
    result.append("weight1")
    result.append("bias1")
    result.append("weight2")
    result.append("bias2")
    return result^


def _mlp_dims(index: Int) -> List[Int]:
    var result = List[Int]()
    if index == 0:
        result.append(16)
        result.append(8)
    elif index == 1:
        result.append(16)
    elif index == 2:
        result.append(3)
        result.append(16)
    else:
        result.append(3)
    return result^
