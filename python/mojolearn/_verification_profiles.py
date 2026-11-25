"""Routine verification includes classical fits; neural training is opt-in."""

NEURAL_FAMILIES = frozenset(('byte_lm', 'neural', 'training', 'mamba',
                             'transformer', 'embedding', 'x_cnn'))
CLASSICAL_SEQUENCE = frozenset(('sequence-autoarima', 'sequence-stl', 'sequence-var',
                                'sequence-theta', 'sequence-croston', 'sequence-ets',
                                'sequence-garch', 'sequence-prophet'))
# One seeded forward, backward and optimizer step through the autograd tape
# (tools/identity_lanes/sequence.py): 64 rows, seconds, so routine `verify`
# covers the tape while full neural training stays opt-in.
ROUTINE_TAPE = frozenset(('sequence-tape-sequential', 'sequence-tape-program',
                          # lane seq-blocks-3: the sequence-block baselines, one seeded step on 32 sequences
                          'sequence-blocks-cnn1d', 'sequence-blocks-resnet1d', 'sequence-blocks-inceptiontime',
                          'sequence-blocks-gru-lengths', 'sequence-blocks-lstm-lengths',

                          # lane neural-blocks-3: composed models, one seeded step each
                          'sequence-composed-transformer', 'sequence-composed-mamba1',
                          'sequence-composed-mamba2', 'sequence-composed-mamba3',
                          'sequence-composed-recurrent',
                          'sequence-composed-conv1d',
                          # lane nn-losses: one seeded step per added loss
                          'sequence-composed-losses',
                          # lane nn-losses-2: log_softmax with nll / kl_div, and ctc
                          'sequence-composed-log-softmax-ctc',
                          # nn-recurrent: the recurrent options (tools/identity_lanes/nn_recurrent.py)
                          'sequence-blocks-bigru-lengths', 'sequence-blocks-bilstm',

                          # nn-attention begin: classic attention models, one seeded step each
                          'sequence-attention-encoder', 'sequence-attention-cross',
                          # nn-attention end
                          # nn-conv begin: the convolution and shape layers, one seeded step each
                          'sequence-nnconv-transpose', 'sequence-nnconv-pools', 'sequence-nnconv-pads',
                          'sequence-nnconv-upsample', 'sequence-nnconv-shuffle',
                          # nn-conv end
                          # nn-activations begin
                          'sequence-composed-activations',
                          # nn-activations end
                          # nn-norms begin: group and instance norm models, one seeded step each
                          'sequence-norms-resnet1d-gn', 'sequence-norms-instance',
                          # nn-norms end
                          # lane nn-optim begin: Sequential.compile's optimizers and schedules, small fits
                          'sequence-optim-rmsprop', 'sequence-optim-adagrad', 'sequence-optim-lamb',
                          # lane nn-optim end
                          # nn-conv3d begin
                          'sequence-composed-conv3d', 'sequence-composed-pool3d',
                          # nn-conv3d end
                          # nn-blocks-scatter begin: ops.Session scatter_add and segment reductions, one seeded backward
                          'ops-scatter-segment',
                          # nn-blocks-scatter end
                          # nn-blocks-scan begin: Session.scan over a captured cell, one seeded step
                          'ops-scan-cell',
                          # nn-blocks-scan end
                          # nn-blocks-cbw begin: a user-defined backward rule (Session.custom_op), seconds
                          'nn-cbw-custom-vjp',
                          # nn-blocks-cbw end
                          # nn-blocks-composed begin: the composed tagger, two steps and a resume (eager, Program.fit)
                          'ops-composed-tagger',
                          # nn-blocks-composed end
                          # session-ops-conv begin: ops.Session conv1d / conv2d dilated / conv3d / conv_transpose, one seeded backward
                          'ops-conv-layers',
                          # session-ops-conv end
                          ))
# These fixtures use loaded weights or inspect configuration without training.
NEURAL_INFERENCE = frozenset(('hf-causal-lm', 'byte-lm-host-infer',
                              'byte-lm-host-infer-threaded', 'language-model-config',
                              'hf-causal-lm-stream',
                              # lane verify-coverage-owed (2026-10-21): seeded weights,
                              # decode or forward only, no training
                              'mamba2-decode-session', 'mamba3-decode-session',
                              'training-mlp-sessions',
                              # lane verify-gaps-20261022: loaded weights, decode only
                              'hf-causal-lm-text'))


def neural_training_lanes(harness, surface):
    lanes = set()
    for family in surface.FAMILIES:
        declared = set(family['training_lanes'])
        if family['family'] in NEURAL_FAMILIES:
            lanes.update(declared)
        elif family['family'] == 'x_sequence':
            lanes.update(declared - CLASSICAL_SEQUENCE)
    return lanes.intersection(harness.LANES) - NEURAL_INFERENCE - ROUTINE_TAPE


def select(harness, surface, lanes, *, neural=False, asked=()):
    training = neural_training_lanes(harness, surface)
    allowed = training if neural else set(harness.LANES) - training
    wrong = set(asked) - allowed
    if wrong:
        hint = 'omit --neural-training for routine checks' if neural else 'use --neural-training explicitly'
        raise ValueError(f'lanes outside this verification profile: {sorted(wrong)}; {hint}')
    selected = [lane for lane in lanes if lane in allowed]
    return selected, dict(
        name='neural-training' if neural else 'routine',
        description=('neural training and backward/optimizer fixtures' if neural else
                     'classical training and inference, plus bundled saved-model inference; neural training excluded'),
        selected_lanes=selected,
        excluded_neural_training=[] if neural else sorted(training),
        neural_training_command='python -m mojolearn verify --neural-training',
        loaded_neural_inference_command='python -m mojolearn verify-causal-lm')
