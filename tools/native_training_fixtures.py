"""External training qualification fixtures; never imported by product code."""
import numpy as np


def fixture(name, epochs=4):
    # options: loss,target,opt,flags,schedule,step,batch,epochs,shuffle,seedlo,seedhi,patience,init
    options = [0, 0, 1, 0, 1, 2, 3, epochs, 1, 873, 0, 0, 1]
    scalars = [.03125, .9, .99, 1e-8, .01, 0., 0., .5, 1.]
    if name == "analytic_sgd":
        model = [1, 1, 1, 1, 1, 1, 0]
        options = [0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0]
        scalars = [.125, 0., 0., 1e-8, 0., 0., 0., 1., 1., 0.]
        arrays = [np.array([[1.], [2.]], np.float32), np.array([[2.], [4.]], np.float32), np.zeros(2, np.float32)]
        return arrays, model, options, scalars
    x = ((np.arange(21, dtype=np.float32).reshape(7, 3) % 11) - 5) / 8
    y = np.stack((x[:, 0] + x[:, 1], x[:, 2] - x[:, 0]), axis=1)
    model = [1, 3, 4, 1, 4, 1, 0, 2, 0, 0, 0, 7, 0, 0, 0, 1, 2, 1, 0]
    layer_floats = [0., 0., .25, 0.]
    nparams = 26
    if name == "adamw_ce":
        options[0:3] = [1, 1, 2]
        y = (np.arange(7) % 2).astype(np.int32)
    elif name == "momentum_huber":
        options[0], options[2], options[3] = 3, 0, 1
        scalars[1] = .75
    elif name == "bce":
        options[0] = 2
        y = (y > 0).astype(np.float32)
    elif name == "rankn_conv":
        # (C,H,W) -> conv2d -> flatten -> linear; version 2 native grammar.
        model = [2, 3, 1, 3, 3, 1, 1, 1, 1, 1, 3,
                 12, 1, 2, 1, 0, 1, 1, 0,
                 11, 0, 0, 0, 0, 0, 0, 0,
                 1, 2, 1, 0, 0, 0, 0, 0]
        x = (np.arange(63, dtype=np.float32).reshape(7, 1, 3, 3) % 13 - 6) / 8
        layer_floats, nparams = [0., 0., 0.], 15
    elif name != "adam_dropout":
        raise ValueError(name)
    return [x, y, np.zeros(nparams, np.float32)], model, options, scalars + layer_floats


CASES = ("analytic_sgd", "adam_dropout", "adamw_ce", "momentum_huber", "bce", "rankn_conv")


def identical(actual, expected):
    assert len(actual) == len(expected)
    for slot, (a, b) in enumerate(zip(actual, expected)):
        assert a.shape == b.shape and a.dtype == b.dtype, (slot, a.shape, b.shape, a.dtype, b.dtype)
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), (slot, a, b)


def analytic(outputs):
    # Independent calculus: MSE=10, gradients (weight,bias)=(-10,-6), lr=1/8.
    np.testing.assert_array_equal(outputs[0], np.array([1.25, .75], np.float32))
    np.testing.assert_array_equal(outputs[4], np.array([10.], np.float32))
    assert outputs[3].view(np.int64)[1:3].tolist() == [1, 1]
