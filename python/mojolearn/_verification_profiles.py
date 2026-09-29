"""Routine verification includes classical fits; neural training is opt-in."""

NEURAL_FAMILIES = frozenset(('byte_lm', 'neural', 'training', 'mamba',
                             'transformer', 'embedding', 'x_cnn'))
CLASSICAL_SEQUENCE = frozenset(('sequence-autoarima', 'sequence-stl', 'sequence-var',
                                'sequence-theta', 'sequence-croston', 'sequence-ets',
                                'sequence-garch', 'sequence-prophet'))
# These fixtures use loaded weights or inspect configuration without training.
NEURAL_INFERENCE = frozenset(('hf-causal-lm', 'byte-lm-host-infer',
                              'byte-lm-host-infer-threaded', 'language-model-config'))


def neural_training_lanes(harness, surface):
    lanes = set()
    for family in surface.FAMILIES:
        declared = set(family['training_lanes'])
        if family['family'] in NEURAL_FAMILIES:
            lanes.update(declared)
        elif family['family'] == 'x_sequence':
            lanes.update(declared - CLASSICAL_SEQUENCE)
    return lanes.intersection(harness.LANES) - NEURAL_INFERENCE


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
