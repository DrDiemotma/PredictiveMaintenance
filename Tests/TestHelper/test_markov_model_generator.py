import numpy as np
from Tests.Helper import MarkovModelGenerator


def test_markov_model_generator():
    transition_matrix: np.typing.NDArray = np.array([[0.8, 0.2, 0.0],
                                                     [0.1, 0.8, 0.1],
                                                     [0.3, 0.0, 0.7]],
                                                    dtype=np.float32)
    init_state: int = 0
    rng: np.random.Generator = np.random.default_rng(42)
    batchsize = 5
    sequence = 15
    dimension = 6
    sut: MarkovModelGenerator = MarkovModelGenerator(transition_matrix, init_state)
    data: np.typing.NDArray = sut.get_next(batchsize, sequence, dimension, rng)
    assert data.shape == (batchsize, sequence, dimension), \
        f"Shape was {data.shape}, expected {(batchsize, sequence, dimension)}."
    total_abs_sum: float = np.sum(np.abs(data))
    assert total_abs_sum > 0, "Generated data is 0."
    assert np.isfinite(data).all(), "Generated data contains NaN or Inf variables."
    state_sequence = sut.states
    assert len(state_sequence) > 0
    assert sum((1 if x is init_state else 0 for x in state_sequence)) < len(state_sequence), \
        "No state transitions from 0."
