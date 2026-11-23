# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the 1-D sequence blocks and the residual_block / parallel containers
(lane seq-blocks, 2026-11-22): the flat description each call writes and the refusals at add.
The numerics (forward, gradients, the composed baselines, host versus device words) are the
native check checks/native_tests/sequence_blocks.mojo."""
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def test_one_d_layers_describe_their_arguments():
    m = (mojolearn.Sequential(input_shape=(2, 16)).conv1d(4, 3, padding=1).max_pool1d(3, stride=1, padding=1)
         .avg_pool1d(2).adaptive_avg_pool1d(2).flatten().linear(2))
    assert _kinds(m) == ["conv1d", "max_pool1d", "avg_pool1d", "adaptive_avg_pool1d", "flatten", "linear"]
    assert m.layers[0][1] == (4, 3, 1, 1, 1, 1)
    assert m.layers[1][1] == (3, 1, 1, 0, 0, 0)
    assert m.layers[2][1] == (2, 2, 0, 0, 0, 0)
    assert m.layers[3][1] == (2, 0, 0, 0, 0, 0)


def test_residual_block_with_shortcut_is_flattened():
    body = mojolearn.Sequential().conv1d(8, 3, padding=1).relu().conv1d(8, 3, padding=1)
    shortcut = mojolearn.Sequential().conv1d(8, 1)
    m = mojolearn.Sequential(input_shape=(2, 16)).conv1d(4, 1).residual_block(body, shortcut).relu()
    assert _kinds(m) == ["conv1d", "conv1d", "relu", "conv1d", "branch", "conv1d", "residual", "relu"]
    assert m.layers[4][1][0] == 0      # the branch returns to the block input (layer 0)
    assert m.layers[6][1] == 3         # the residual adds the body's last layer
    assert len(body.layers) == 3 and len(shortcut.layers) == 1


def test_identity_residual_block_adds_the_block_input():
    body = mojolearn.Sequential().conv1d(4, 3, padding=1)
    m = mojolearn.Sequential(input_shape=(4, 16)).residual_block(body)
    assert _kinds(m) == ["conv1d", "residual"]
    assert m.layers[1][1] == -1        # the block input is the model input


def test_parallel_concatenates_branch_ends():
    a = mojolearn.Sequential().conv1d(4, 9, padding=4)
    b = mojolearn.Sequential().conv1d(4, 5, padding=2)
    c = mojolearn.Sequential().max_pool1d(3, stride=1, padding=1).conv1d(4, 1)
    m = mojolearn.Sequential(input_shape=(2, 16)).conv1d(3, 1).parallel(a, b, c)
    assert _kinds(m) == ["conv1d", "conv1d", "branch", "conv1d", "branch", "max_pool1d", "conv1d", "concat"]
    assert m.layers[7][1] == (1, 3, 6, -2, -2, -2)


def test_nested_references_are_renumbered():
    inner = mojolearn.Sequential().conv1d(4, 3, padding=1).residual(-1)
    m = mojolearn.Sequential(input_shape=(4, 16)).relu().parallel(inner, mojolearn.Sequential())
    # the inner residual's -1 (its own input) becomes the block input, layer 0
    assert m.layers[2] == ("residual", 0, 0, 0.0)
    assert m.layers[3][1] == (2, 0, -2, -2, -2, -2)


def test_container_refusals():
    m = mojolearn.Sequential(input_shape=(2, 16))
    with pytest.raises(Exception, match="Sequential"):
        m.residual_block("conv")
    with pytest.raises(Exception, match="at least one layer"):
        m.residual_block(mojolearn.Sequential())
    with pytest.raises(Exception, match="1 to 6 branches"):
        m.parallel(*[mojolearn.Sequential().relu() for _ in range(7)])
    with pytest.raises(Exception, match="half of kernel_size"):
        m.max_pool1d(2, padding=2)


def test_selective_scan_and_feature_concat_describe_their_arguments():
    # lane neural-blocks-2: Mamba-1's core as a layer; its [u | delta | B | C] input joined on the features
    m = (mojolearn.Sequential(input_shape=(6, 4)).linear(3).branch(-1).linear(3).softplus().branch(-1)
         .linear(8).concat([0, 3, 5], axis=1).selective_scan(4))
    assert _kinds(m)[-2:] == ["concat", "selective_scan"]
    assert m.layers[-1][1] == (4, 0, 0, 0, 0, 0)
    assert 1.0 in m.layers[-2]        # the joined sample axis rides in the layer's float argument
    with pytest.raises(Exception, match="d_state must be 4, 8 or 16"):
        mojolearn.Sequential(input_shape=(6, 14)).selective_scan(5)
