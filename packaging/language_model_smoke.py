"""Installed native generalized/resident training gate; no speed or quality claim."""
import numpy as np
from mojolearn import LanguageModelConfig, LanguageModelTrainer


def main():
    shape = LanguageModelConfig(batch=1, length=5, d_model=16, n_heads=2,
                                n_kv=1, head_dim=8, intermediate=24,
                                n_layers=3, vocab_size=257)
    rng = np.random.default_rng(19)
    weights = (rng.standard_normal(shape.n_total) * .02).astype(np.float32)
    ids = rng.integers(0, shape.vocab_size, (shape.batch, shape.length + 1), dtype=np.int32)
    trainers = [LanguageModelTrainer(weights, shape=shape, resident=resident,
                                    data_schedule={'dataset': 'wheel-smoke'})
                for resident in (False, True)]
    try:
        for _ in range(2):
            results = [trainer.train_step(ids) for trainer in trainers]
            assert results[0]['loss'] == results[1]['loss']
            states = [trainer.state_dict() for trainer in trainers]
            for key, value in states[0].items():
                other = states[1][key]
                if isinstance(value, np.ndarray):
                    assert value.dtype == other.dtype and value.shape == other.shape
                    assert value.tobytes() == other.tobytes(), key
                else:
                    assert value == other, key
        assert trainers[0].evaluate(ids) == trainers[1].evaluate(ids)
        assert trainers[0].step_ == trainers[1].step_ == 2
    finally:
        for trainer in trainers:
            trainer.close()
    print('PASS generalized three-layer/vocab257 resident and stateless training')


if __name__ == '__main__':
    main()
