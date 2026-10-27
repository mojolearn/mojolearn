import struct
import bench_board_neural as N
import bench_board_params as P


def test_fit_arms_share_exact_fp32_rate_without_relaxing_parameter_equality():
    rate = N.FIT_ADAM['lr']
    assert rate == struct.unpack('f', struct.pack('f', 0.001))[0]
    for lane in N.FIT_LANES:
        assert N.fit_params(lane)['learning_rate'] == rate
    assert P._equal(rate, struct.unpack('f', struct.pack('f', rate))[0])
    assert not P._equal(rate, 0.001)
    assert not P._equal(rate, 0.002)
